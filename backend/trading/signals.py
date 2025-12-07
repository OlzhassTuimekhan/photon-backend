from decimal import Decimal
from django.db.models.signals import post_save
from django.dispatch import receiver
from django.contrib.auth import get_user_model

from trading.models import AgentStatus, Account


User = get_user_model()


@receiver(post_save, sender=User)
def create_default_agent_statuses(sender, instance, created, **kwargs):
    """Создаем статусы агентов и счет при создании пользователя"""
    if created:
        for agent_type, _ in AgentStatus.AGENT_TYPES:
            AgentStatus.objects.get_or_create(
                user=instance,
                agent_type=agent_type,
                defaults={"status": "IDLE"},
            )
        # Создаем счет пользователя с начальным балансом
        Account.objects.get_or_create(
            user=instance,
            defaults={
                "balance": Decimal("10000.00"),
                "free_cash": Decimal("10000.00"),
                "initial_balance": Decimal("10000.00"),
            }
        )

